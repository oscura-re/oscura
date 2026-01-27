"""Core data types for Oscura signal analysis framework.

This module implements the fundamental data structures for oscilloscope
and logic analyzer data analysis.

Requirements addressed:
- CORE-001: TraceMetadata Data Class
- CORE-002: WaveformTrace Data Class
- CORE-003: DigitalTrace Data Class
- CORE-004: ProtocolPacket Data Class
- CORE-005: CalibrationInfo Data Class (regulatory compliance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray


@dataclass
class CalibrationInfo:
    """Calibration and instrument provenance information.

    Stores traceability metadata for measurements performed on oscilloscope
    or logic analyzer data. Essential for regulatory compliance and quality
    assurance in DOD/aerospace/medical applications.

    Attributes:
        instrument: Instrument make and model (e.g., "Tektronix DPO7254C").
        serial_number: Instrument serial number for traceability (optional).
        calibration_date: Date of last calibration (optional).
        calibration_due_date: Date when next calibration is due (optional).
        firmware_version: Instrument firmware version (optional).
        calibration_lab: Calibration lab name or accreditation (optional).
        calibration_cert_number: Calibration certificate number (optional).
        probe_attenuation: Probe attenuation factor (e.g., 10.0 for 10x probe) (optional).
        coupling: Input coupling ("DC", "AC", "GND") (optional).
        bandwidth_limit: Bandwidth limit in Hz, None if disabled (optional).
        vertical_resolution: ADC resolution in bits (optional).
        timebase_accuracy: Timebase accuracy in ppm (parts per million) (optional).

    Example:
        >>> from datetime import datetime
        >>> cal_info = CalibrationInfo(
        ...     instrument="Tektronix DPO7254C",
        ...     serial_number="C012345",
        ...     calibration_date=datetime(2024, 12, 15),
        ...     probe_attenuation=10.0,
        ...     vertical_resolution=8
        ... )
        >>> print(f"Instrument: {cal_info.instrument}")
        Instrument: Tektronix DPO7254C

    References:
        ISO/IEC 17025: General Requirements for Testing/Calibration Laboratories
        NIST Handbook 150: Laboratory Accreditation Program Requirements
        21 CFR Part 11: Electronic Records (FDA)
    """

    instrument: str
    serial_number: str | None = None
    calibration_date: datetime | None = None
    calibration_due_date: datetime | None = None
    firmware_version: str | None = None
    calibration_lab: str | None = None
    calibration_cert_number: str | None = None
    probe_attenuation: float | None = None
    coupling: str | None = None
    bandwidth_limit: float | None = None
    vertical_resolution: int | None = None
    timebase_accuracy: float | None = None

    def __post_init__(self) -> None:
        """Validate calibration info after initialization."""
        if self.probe_attenuation is not None and self.probe_attenuation <= 0:
            raise ValueError(f"probe_attenuation must be positive, got {self.probe_attenuation}")
        if self.bandwidth_limit is not None and self.bandwidth_limit <= 0:
            raise ValueError(f"bandwidth_limit must be positive, got {self.bandwidth_limit}")
        if self.vertical_resolution is not None and self.vertical_resolution <= 0:
            raise ValueError(
                f"vertical_resolution must be positive, got {self.vertical_resolution}"
            )
        if self.timebase_accuracy is not None and self.timebase_accuracy <= 0:
            raise ValueError(f"timebase_accuracy must be positive, got {self.timebase_accuracy}")

    @property
    def is_calibration_current(self) -> bool | None:
        """Check if calibration is current.

        Returns:
            True if calibration is current, False if expired, None if dates not set.
        """
        if self.calibration_date is None or self.calibration_due_date is None:
            return None
        from datetime import datetime

        return datetime.now() < self.calibration_due_date

    @property
    def traceability_summary(self) -> str:
        """Generate a traceability summary string.

        Returns:
            Human-readable summary of calibration traceability.
        """
        parts = [f"Instrument: {self.instrument}"]
        if self.serial_number:
            parts.append(f"S/N: {self.serial_number}")
        if self.calibration_date:
            parts.append(f"Cal Date: {self.calibration_date.strftime('%Y-%m-%d')}")
        if self.calibration_due_date:
            parts.append(f"Due: {self.calibration_due_date.strftime('%Y-%m-%d')}")
        if self.calibration_cert_number:
            parts.append(f"Cert: {self.calibration_cert_number}")
        return ", ".join(parts)


@dataclass
class TraceMetadata:
    """Metadata describing a captured trace.

    Contains sample rate, scaling information, acquisition details,
    and provenance information for a captured waveform or digital trace.

    Attributes:
        sample_rate: Sample rate in Hz (required).
        vertical_scale: Vertical scale in volts/division (optional).
        vertical_offset: Vertical offset in volts (optional).
        acquisition_time: Time of acquisition (optional).
        trigger_info: Trigger configuration dictionary (optional).
        source_file: Path to source file (optional).
        channel_name: Name of the channel (optional).
        calibration_info: Calibration and instrument traceability information (optional).

    Example:
        >>> metadata = TraceMetadata(sample_rate=1e9)  # 1 GSa/s
        >>> print(f"Time base: {metadata.time_base} s/sample")
        Time base: 1e-09 s/sample

    Example with calibration info:
        >>> from datetime import datetime
        >>> cal = CalibrationInfo(
        ...     instrument="Tektronix DPO7254C",
        ...     calibration_date=datetime(2024, 12, 15)
        ... )
        >>> metadata = TraceMetadata(sample_rate=1e9, calibration_info=cal)
        >>> print(metadata.calibration_info.traceability_summary)
        Instrument: Tektronix DPO7254C, Cal Date: 2024-12-15

    References:
        IEEE 181-2011: Standard for Transitional Waveform Definitions
        ISO/IEC 17025: General Requirements for Testing/Calibration Laboratories
    """

    sample_rate: float
    vertical_scale: float | None = None
    vertical_offset: float | None = None
    acquisition_time: datetime | None = None
    trigger_info: dict[str, Any] | None = None
    source_file: str | None = None
    channel_name: str | None = None
    calibration_info: CalibrationInfo | None = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def time_base(self) -> float:
        """Time between samples in seconds (derived from sample_rate).

        Returns:
            Time per sample in seconds (1 / sample_rate).
        """
        return 1.0 / self.sample_rate


@dataclass
class WaveformTrace:
    """Analog waveform data with metadata.

    Stores sampled analog voltage data as a numpy array along with
    associated metadata for timing and scaling.

    Attributes:
        data: Waveform samples as numpy float array.
        metadata: Associated trace metadata.

    Example:
        >>> import numpy as np
        >>> data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
        >>> trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
        >>> print(f"Duration: {trace.time_vector[-1]:.6f} seconds")
        Duration: 0.000999 seconds

    References:
        IEEE 1241-2010: Standard for Terminology and Test Methods for ADCs
    """

    data: NDArray[np.floating[Any]]
    metadata: TraceMetadata

    def __post_init__(self) -> None:
        """Validate waveform data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be a numpy array, got {type(self.data).__name__}")
        if not np.issubdtype(self.data.dtype, np.floating):
            # Convert to float64 if not already floating point
            self.data = self.data.astype(np.float64)

    @property
    def time_vector(self) -> NDArray[np.float64]:
        """Time axis in seconds.

        Computes a time vector starting from 0, with intervals
        determined by the sample rate.

        Returns:
            Array of time values in seconds, same length as data.
        """
        n_samples = len(self.data)
        return np.arange(n_samples, dtype=np.float64) * self.metadata.time_base

    @property
    def duration(self) -> float:
        """Total duration of the trace in seconds.

        Returns:
            Duration from first to last sample in seconds.
        """
        if len(self.data) == 0:
            return 0.0
        return (len(self.data) - 1) * self.metadata.time_base

    def __len__(self) -> int:
        """Return number of samples in the trace."""
        return len(self.data)


@dataclass
class DigitalTrace:
    """Digital/logic signal data with metadata.

    Stores sampled digital signal data as a boolean numpy array,
    with optional edge timestamp information.

    Attributes:
        data: Digital samples as numpy boolean array.
        metadata: Associated trace metadata.
        edges: Optional list of (timestamp, is_rising) tuples.

    Example:
        >>> import numpy as np
        >>> data = np.array([False, False, True, True, False], dtype=bool)
        >>> trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
        >>> print(f"High samples: {np.sum(trace.data)}")
        High samples: 2

    References:
        IEEE 1076.6-2004: Standard for VHDL Register Transfer Level Synthesis
    """

    data: NDArray[np.bool_]
    metadata: TraceMetadata
    edges: list[tuple[float, bool]] | None = None

    def __post_init__(self) -> None:
        """Validate digital data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be a numpy array, got {type(self.data).__name__}")
        if self.data.dtype != np.bool_:
            # Convert to boolean if not already
            self.data = self.data.astype(np.bool_)

    @property
    def time_vector(self) -> NDArray[np.float64]:
        """Time axis in seconds.

        Returns:
            Array of time values in seconds, same length as data.
        """
        n_samples = len(self.data)
        return np.arange(n_samples, dtype=np.float64) * self.metadata.time_base

    @property
    def duration(self) -> float:
        """Total duration of the trace in seconds.

        Returns:
            Duration from first to last sample in seconds.
        """
        if len(self.data) == 0:
            return 0.0
        return (len(self.data) - 1) * self.metadata.time_base

    @property
    def rising_edges(self) -> list[float]:
        """Timestamps of rising edges.

        Returns:
            List of timestamps where signal transitions from low to high.
        """
        if self.edges is None:
            return []
        return [ts for ts, is_rising in self.edges if is_rising]

    @property
    def falling_edges(self) -> list[float]:
        """Timestamps of falling edges.

        Returns:
            List of timestamps where signal transitions from high to low.
        """
        if self.edges is None:
            return []
        return [ts for ts, is_rising in self.edges if not is_rising]

    def __len__(self) -> int:
        """Return number of samples in the trace."""
        return len(self.data)


@dataclass
class IQTrace:
    """I/Q (In-phase/Quadrature) waveform data with metadata.

    Stores complex-valued signal data as separate I and Q components,
    commonly used for RF and software-defined radio applications.

    Attributes:
        i_data: In-phase component samples as numpy float array.
        q_data: Quadrature component samples as numpy float array.
        metadata: Associated trace metadata.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 1e-3, 1000)
        >>> i_data = np.cos(2 * np.pi * 1e6 * t)
        >>> q_data = np.sin(2 * np.pi * 1e6 * t)
        >>> trace = IQTrace(i_data=i_data, q_data=q_data, metadata=TraceMetadata(sample_rate=1e6))
        >>> print(f"Complex samples: {len(trace)}")
        Complex samples: 1000

    References:
        IEEE Std 181-2011: Transitional Waveform Definitions
    """

    i_data: NDArray[np.floating[Any]]
    q_data: NDArray[np.floating[Any]]
    metadata: TraceMetadata

    def __post_init__(self) -> None:
        """Validate I/Q data after initialization."""
        if not isinstance(self.i_data, np.ndarray):
            raise TypeError(f"i_data must be a numpy array, got {type(self.i_data).__name__}")
        if not isinstance(self.q_data, np.ndarray):
            raise TypeError(f"q_data must be a numpy array, got {type(self.q_data).__name__}")
        if len(self.i_data) != len(self.q_data):
            raise ValueError(
                f"I and Q data must have same length, got {len(self.i_data)} and {len(self.q_data)}"
            )
        # Convert to float64 if not already floating point
        if not np.issubdtype(self.i_data.dtype, np.floating):
            self.i_data = self.i_data.astype(np.float64)
        if not np.issubdtype(self.q_data.dtype, np.floating):
            self.q_data = self.q_data.astype(np.float64)

    @property
    def complex_data(self) -> NDArray[np.complex128]:
        """Return I/Q data as complex array.

        Returns:
            Complex array where real=I, imag=Q.
        """
        return self.i_data + 1j * self.q_data

    @property
    def magnitude(self) -> NDArray[np.float64]:
        """Magnitude (amplitude) of the complex signal.

        Returns:
            Array of magnitude values sqrt(I² + Q²).
        """
        return np.sqrt(self.i_data**2 + self.q_data**2)

    @property
    def phase(self) -> NDArray[np.float64]:
        """Phase angle of the complex signal in radians.

        Returns:
            Array of phase values atan2(Q, I).
        """
        return np.arctan2(self.q_data, self.i_data)

    @property
    def time_vector(self) -> NDArray[np.float64]:
        """Time axis in seconds.

        Returns:
            Array of time values in seconds, same length as data.
        """
        n_samples = len(self.i_data)
        return np.arange(n_samples, dtype=np.float64) * self.metadata.time_base

    @property
    def duration(self) -> float:
        """Total duration of the trace in seconds.

        Returns:
            Duration from first to last sample in seconds.
        """
        if len(self.i_data) == 0:
            return 0.0
        return (len(self.i_data) - 1) * self.metadata.time_base

    def __len__(self) -> int:
        """Return number of samples in the trace."""
        return len(self.i_data)


@dataclass
class ProtocolPacket:
    """Decoded protocol packet data.

    Represents a decoded packet from a serial protocol (UART, SPI, I2C, etc.)
    with timing, data content, annotations, and error information.

    Attributes:
        timestamp: Start time of the packet in seconds.
        protocol: Name of the protocol (e.g., "UART", "SPI", "I2C").
        data: Decoded data bytes.
        annotations: Multi-level annotations dictionary (optional).
        errors: List of detected errors (optional).
        end_timestamp: End time of the packet in seconds (optional).

    Example:
        >>> packet = ProtocolPacket(
        ...     timestamp=1.23e-3,
        ...     protocol="UART",
        ...     data=b"Hello"
        ... )
        >>> print(f"Received at {packet.timestamp}s: {packet.data.decode()}")
        Received at 0.00123s: Hello

    References:
        sigrok Protocol Decoder API
    """

    timestamp: float
    protocol: str
    data: bytes
    annotations: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    end_timestamp: float | None = None

    def __post_init__(self) -> None:
        """Validate packet data after initialization."""
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if not isinstance(self.data, bytes):
            raise TypeError(f"data must be bytes, got {type(self.data).__name__}")

    @property
    def duration(self) -> float | None:
        """Duration of the packet in seconds.

        Returns:
            Duration if end_timestamp is set, None otherwise.
        """
        if self.end_timestamp is None:
            return None
        return self.end_timestamp - self.timestamp

    @property
    def has_errors(self) -> bool:
        """Check if packet has any errors.

        Returns:
            True if errors list is non-empty.
        """
        return len(self.errors) > 0

    def __len__(self) -> int:
        """Return number of bytes in the packet."""
        return len(self.data)


# Type aliases for convenience
Trace = WaveformTrace | DigitalTrace | IQTrace
"""Union type for any trace type."""

__all__ = [
    "CalibrationInfo",
    "DigitalTrace",
    "IQTrace",
    "ProtocolPacket",
    "Trace",
    "TraceMetadata",
    "WaveformTrace",
]
