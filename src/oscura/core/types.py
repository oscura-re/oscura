"""Core data types for Oscura signal analysis framework.

This module implements the fundamental data structures for oscilloscope
and logic analyzer data analysis.

Requirements addressed:
- CORE-001: TraceMetadata Data Class
- CORE-002: WaveformTrace Data Class
- CORE-003: DigitalTrace Data Class
- CORE-004: ProtocolPacket Data Class
- CORE-005: CalibrationInfo Data Class (regulatory compliance)
- CORE-006: MeasurementResult TypedDict (v0.9.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray


class MeasurementResult(TypedDict, total=False):
    """Structured measurement result with metadata and applicability tracking.

    This replaces raw float values to handle edge cases gracefully and provide
    rich metadata for reporting and interpretation.

    Attributes:
        value: Measurement value (None if not applicable).
        unit: Unit of measurement (e.g., "V", "Hz", "s", "dB", "%", "ratio").
        applicable: Whether measurement is applicable to this signal type.
        reason: Explanation if not applicable (e.g., "Aperiodic signal").
        display: Human-readable formatted display string.

    Example:
        >>> # Applicable measurement
        >>> freq_result: MeasurementResult = {
        ...     "value": 1000.0,
        ...     "unit": "Hz",
        ...     "applicable": True,
        ...     "reason": None,
        ...     "display": "1.000 kHz"
        ... }

        >>> # Inapplicable measurement (no NaN!)
        >>> period_result: MeasurementResult = {
        ...     "value": None,
        ...     "unit": "s",
        ...     "applicable": False,
        ...     "reason": "Aperiodic signal (single impulse)",
        ...     "display": "N/A"
        ... }

        >>> # Access safely
        >>> if period_result["applicable"]:
        ...     print(f"Period: {period_result['display']}")
        ... else:
        ...     print(f"Period: {period_result['display']} ({period_result['reason']})")
        Period: N/A (Aperiodic signal (single impulse))

    References:
        API Improvement Recommendation #3 (v0.9.0)
    """

    value: float | None
    unit: str
    applicable: bool
    reason: str | None
    display: str


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

        now = datetime.now(UTC)
        # Ensure dates are timezone-aware for comparison
        due_date = self.calibration_due_date
        if due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=UTC)

        return now < due_date


@dataclass
class TraceMetadata:
    """Metadata for waveform and digital traces.

    Stores acquisition parameters, channel information, and optional
    calibration data for oscilloscope/logic analyzer captures.

    Attributes:
        sample_rate: Sample rate in Hz.
        start_time: Start time in seconds (relative to trigger).
        channel: Channel name or number.
        units: Physical units (e.g., "V", "A", "Pa").
        calibration: Optional calibration and provenance information.
        trigger_time: Trigger timestamp in seconds (optional).
        coupling: Input coupling mode ("DC", "AC", "GND") (optional).
        probe_attenuation: Probe attenuation factor (optional).
        bandwidth_limit: Bandwidth limit in Hz (optional).
        vertical_offset: Vertical offset in physical units (optional).
        vertical_scale: Vertical scale in units/division (optional).
        horizontal_scale: Horizontal scale in seconds/division (optional).
        source_file: Source file path for loaded data (optional).
        trigger_info: Additional trigger metadata as dict (optional).
        acquisition_time: Timestamp when data was acquired (optional).

    Example:
        >>> meta = TraceMetadata(
        ...     sample_rate=1e9,
        ...     start_time=-0.001,
        ...     channel="CH1",
        ...     units="V"
        ... )
        >>> print(f"Sample rate: {meta.sample_rate/1e6:.1f} MS/s")
        Sample rate: 1000.0 MS/s
    """

    sample_rate: float
    start_time: float = 0.0
    channel: str = "CH1"
    units: str = "V"
    calibration: CalibrationInfo | None = None
    trigger_time: float | None = None
    coupling: str | None = None
    probe_attenuation: float | None = None
    bandwidth_limit: float | None = None
    vertical_offset: float | None = None
    vertical_scale: float | None = None
    horizontal_scale: float | None = None
    source_file: str | None = None
    trigger_info: dict[str, Any] | None = None
    acquisition_time: datetime | None = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")


@dataclass
class WaveformTrace:
    """Analog waveform trace with metadata.

    Represents time-series voltage/current data from an oscilloscope or
    similar instrument. Provides properties for signal type detection.

    Attributes:
        data: Waveform data array (voltage/current values).
        metadata: Trace metadata (sample rate, channel, units).

    Properties:
        is_analog: Always True for WaveformTrace.
        is_digital: Always False for WaveformTrace.
        is_iq: Always False for WaveformTrace.
        signal_type: Returns "analog".

    Example:
        >>> import numpy as np
        >>> data = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.001, 1000))
        >>> meta = TraceMetadata(sample_rate=1e6, units="V")
        >>> trace = WaveformTrace(data=data, metadata=meta)
        >>> print(f"Signal type: {trace.signal_type}")
        Signal type: analog
        >>> print(f"Is analog: {trace.is_analog}")
        Is analog: True
    """

    data: NDArray[np.floating[Any]]
    metadata: TraceMetadata

    def __post_init__(self) -> None:
        """Validate trace data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(self.data).__name__}")
        if self.data.ndim != 1:
            raise ValueError(f"data must be 1-D array, got shape {self.data.shape}")
        if len(self.data) == 0:
            raise ValueError("data array cannot be empty")

    @property
    def duration(self) -> float:
        """Duration of the trace in seconds (time from first to last sample)."""
        if len(self.data) <= 1:
            return 0.0
        return (len(self.data) - 1) / self.metadata.sample_rate

    @property
    def time(self) -> NDArray[np.floating[Any]]:
        """Time axis array for the trace."""
        return np.arange(len(self.data)) / self.metadata.sample_rate + self.metadata.start_time

    @property
    def is_analog(self) -> bool:
        """Check if trace is analog (always True for WaveformTrace)."""
        return True

    @property
    def is_digital(self) -> bool:
        """Check if trace is digital (always False for WaveformTrace)."""
        return False

    @property
    def is_iq(self) -> bool:
        """Check if trace is I/Q data (always False for WaveformTrace)."""
        return False

    @property
    def signal_type(self) -> str:
        """Get signal type string (always 'analog' for WaveformTrace)."""
        return "analog"

    @property
    def is_analog(self) -> bool:
        """Check if this is an analog signal trace.

        Returns:
            True for WaveformTrace (always analog).
        """
        return True

    @property
    def is_digital(self) -> bool:
        """Check if this is a digital signal trace.

        Returns:
            False for WaveformTrace (always analog).
        """
        return False

    @property
    def is_iq(self) -> bool:
        """Check if this is an I/Q signal trace.

        Returns:
            False for WaveformTrace.
        """
        return False

    @property
    def signal_type(self) -> str:
        """Get the signal type identifier.

        Returns:
            "analog" for WaveformTrace.
        """
        return "analog"

    def __len__(self) -> int:
        """Return number of samples in the trace."""
        return len(self.data)

    def __getitem__(self, key: int | slice) -> float | NDArray[np.floating[Any]]:
        """Get sample(s) by index."""
        return self.data[key]


@dataclass
class DigitalTrace:
    """Digital logic trace with metadata.

    Represents binary logic level data from a logic analyzer or
    digital channel. Provides properties for signal type detection.

    Attributes:
        data: Boolean array representing logic levels.
        metadata: Trace metadata (sample rate, channel, units).

    Properties:
        is_analog: Always False for DigitalTrace.
        is_digital: Always True for DigitalTrace.
        is_iq: Always False for DigitalTrace.
        signal_type: Returns "digital".

    Example:
        >>> data = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
        >>> meta = TraceMetadata(sample_rate=1e6, units="logic")
        >>> trace = DigitalTrace(data=data, metadata=meta)
        >>> print(f"Signal type: {trace.signal_type}")
        Signal type: digital
        >>> print(f"Is digital: {trace.is_digital}")
        Is digital: True
    """

    data: NDArray[np.bool_]
    metadata: TraceMetadata

    def __post_init__(self) -> None:
        """Validate trace data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(self.data).__name__}")
        if self.data.dtype != bool:
            raise TypeError(f"data must be boolean array, got dtype {self.data.dtype}")
        if self.data.ndim != 1:
            raise ValueError(f"data must be 1-D array, got shape {self.data.shape}")
        if len(self.data) == 0:
            raise ValueError("data array cannot be empty")

    @property
    def duration(self) -> float:
        """Duration of the trace in seconds (time from first to last sample)."""
        if len(self.data) <= 1:
            return 0.0
        return (len(self.data) - 1) / self.metadata.sample_rate

    @property
    def time(self) -> NDArray[np.floating[Any]]:
        """Time axis array for the trace."""
        return np.arange(len(self.data)) / self.metadata.sample_rate + self.metadata.start_time

    @property
    def is_analog(self) -> bool:
        """Check if trace is analog (always False for DigitalTrace)."""
        return False

    @property
    def is_digital(self) -> bool:
        """Check if trace is digital (always True for DigitalTrace)."""
        return True

    @property
    def is_iq(self) -> bool:
        """Check if trace is I/Q data (always False for DigitalTrace)."""
        return False

    @property
    def signal_type(self) -> str:
        """Get signal type string (always 'digital' for DigitalTrace)."""
        return "digital"

    @property
    def is_analog(self) -> bool:
        """Check if this is an analog signal trace.

        Returns:
            False for DigitalTrace (always digital).
        """
        return False

    @property
    def is_digital(self) -> bool:
        """Check if this is a digital signal trace.

        Returns:
            True for DigitalTrace (always digital).
        """
        return True

    @property
    def is_iq(self) -> bool:
        """Check if this is an I/Q signal trace.

        Returns:
            False for DigitalTrace.
        """
        return False

    @property
    def signal_type(self) -> str:
        """Get the signal type identifier.

        Returns:
            "digital" for DigitalTrace.
        """
        return "digital"

    def __len__(self) -> int:
        """Return number of samples in the trace."""
        return len(self.data)

    def __getitem__(self, key: int | slice) -> bool | NDArray[np.bool_]:
        """Get sample(s) by index."""
        return self.data[key]


@dataclass
class IQTrace:
    """I/Q (complex) trace for RF/SDR applications.

    Represents complex-valued I/Q data from software-defined radios
    or RF measurement equipment. Provides properties for signal type detection.

    Attributes:
        data: Complex-valued I/Q data array.
        metadata: Trace metadata (sample rate, channel, units).

    Properties:
        is_analog: Always False for IQTrace.
        is_digital: Always False for IQTrace.
        is_iq: Always True for IQTrace.
        signal_type: Returns "iq".

    Example:
        >>> data = np.exp(1j * 2 * np.pi * np.linspace(0, 1, 100))
        >>> meta = TraceMetadata(sample_rate=1e6, units="V")
        >>> trace = IQTrace(data=data, metadata=meta)
        >>> print(f"Signal type: {trace.signal_type}")
        Signal type: iq
        >>> print(f"Is I/Q: {trace.is_iq}")
        Is I/Q: True
    """

    data: NDArray[np.complexfloating[Any, Any]]
    metadata: TraceMetadata

    def __post_init__(self) -> None:
        """Validate trace data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(self.data).__name__}")
        if not np.iscomplexobj(self.data):
            raise TypeError(f"data must be complex array, got dtype {self.data.dtype}")
        if self.data.ndim != 1:
            raise ValueError(f"data must be 1-D array, got shape {self.data.shape}")
        if len(self.data) == 0:
            raise ValueError("data array cannot be empty")

    @property
    def duration(self) -> float:
        """Duration of the trace in seconds (time from first to last sample)."""
        if len(self.data) <= 1:
            return 0.0
        return (len(self.data) - 1) / self.metadata.sample_rate

    @property
    def time(self) -> NDArray[np.floating[Any]]:
        """Time axis array for the trace."""
        return np.arange(len(self.data)) / self.metadata.sample_rate + self.metadata.start_time

    @property
    def is_analog(self) -> bool:
        """Check if trace is analog (always False for IQTrace)."""
        return False

    @property
    def is_digital(self) -> bool:
        """Check if trace is digital (always False for IQTrace)."""
        return False

    @property
    def is_iq(self) -> bool:
        """Check if trace is I/Q data (always True for IQTrace)."""
        return True

    @property
    def signal_type(self) -> str:
        """Get signal type string (always 'iq' for IQTrace)."""
        return "iq"

    @property
    def is_analog(self) -> bool:
        """Check if this is an analog signal trace.

        Returns:
            False for IQTrace (complex I/Q data).
        """
        return False

    @property
    def is_digital(self) -> bool:
        """Check if this is a digital signal trace.

        Returns:
            False for IQTrace (complex I/Q data).
        """
        return False

    @property
    def is_iq(self) -> bool:
        """Check if this is an I/Q signal trace.

        Returns:
            True for IQTrace (always I/Q).
        """
        return True

    @property
    def signal_type(self) -> str:
        """Get the signal type identifier.

        Returns:
            "iq" for IQTrace.
        """
        return "iq"

    def __len__(self) -> int:
        """Return number of samples in the trace."""
        return len(self.data)

    def __getitem__(self, key: int | slice) -> complex | NDArray[np.complexfloating[Any, Any]]:
        """Get sample(s) by index."""
        return self.data[key]


@dataclass
class ProtocolPacket:
    """Decoded protocol packet with metadata.

    Represents a decoded packet/frame from protocol analysis
    (UART, SPI, I2C, CAN, etc.).

    Attributes:
        timestamp: Packet timestamp in seconds.
        protocol: Protocol name (e.g., "UART", "SPI", "I2C").
        data: Raw packet data as bytes.
        annotations: Protocol-specific annotations (e.g., address, command).
        errors: List of decoding errors (empty if no errors).
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
    "MeasurementResult",
    "ProtocolPacket",
    "Trace",
    "TraceMetadata",
    "WaveformTrace",
]
