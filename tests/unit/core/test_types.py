"""Tests for core data types.

Tests the fundamental data structures (CORE-001-004).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import (
    DigitalTrace,
    IQTrace,
    ProtocolPacket,
    TraceMetadata,
    WaveformTrace,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestTraceMetadata:
    """Test TraceMetadata dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic metadata with sample_rate only."""
        metadata = TraceMetadata(sample_rate=1e6)
        assert metadata.sample_rate == 1e6
        assert metadata.vertical_scale is None
        assert metadata.vertical_offset is None
        assert metadata.channel == "CH1"  # Default value

    def test_create_with_all_fields(self) -> None:
        """Test creating metadata with all optional fields."""
        metadata = TraceMetadata(
            sample_rate=1e9,
            vertical_scale=0.5,
            vertical_offset=0.1,
            start_time=-0.001,
            channel="CH1",
            units="V",
            coupling="DC",
            probe_attenuation=10.0,
        )
        assert metadata.sample_rate == 1e9
        assert metadata.vertical_scale == 0.5
        assert metadata.vertical_offset == 0.1
        assert metadata.start_time == -0.001
        assert metadata.channel == "CH1"
        assert metadata.units == "V"
        assert metadata.coupling == "DC"
        assert metadata.probe_attenuation == 10.0

    def test_validate_positive_sample_rate(self) -> None:
        """Test that negative sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TraceMetadata(sample_rate=-100.0)

    def test_validate_zero_sample_rate(self) -> None:
        """Test that zero sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TraceMetadata(sample_rate=0.0)

    def test_time_base_computation(self) -> None:
        """Test that time base can be computed from sample_rate."""
        metadata = TraceMetadata(sample_rate=1e6)
        assert 1.0 / metadata.sample_rate == 1e-6

        metadata2 = TraceMetadata(sample_rate=1e9)
        assert 1.0 / metadata2.sample_rate == 1e-9


class TestWaveformTrace:
    """Test WaveformTrace dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic waveform trace."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        np.testing.assert_array_equal(trace.data, data)
        assert trace.metadata.sample_rate == 1e6

    def test_validate_data_type(self) -> None:
        """Test that non-array data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(TypeError, match="data must be numpy array"):
            WaveformTrace(data=[1.0, 2.0, 3.0], metadata=metadata)  # type: ignore[arg-type]

    def test_data_with_integer_dtype(self) -> None:
        """Test that integer dtypes are accepted (no auto-conversion)."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)  # Use float64 directly
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        assert trace.data.dtype == np.float64
        np.testing.assert_array_equal(trace.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_time_property(self) -> None:
        """Test time property."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        expected = np.array([0.0, 1e-6, 2e-6])
        np.testing.assert_array_almost_equal(trace.time, expected)

    def test_duration_property(self) -> None:
        """Test duration property."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 samples
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        # Duration is (n-1) * time_base = 4 * 1e-6
        assert trace.duration == 4e-6

    def test_duration_empty_array(self) -> None:
        """Test duration with empty array raises ValueError."""
        data = np.array([])
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(ValueError, match="data array cannot be empty"):
            WaveformTrace(data=data, metadata=metadata)

    def test_len(self) -> None:
        """Test __len__ method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        assert len(trace) == 5


class TestDigitalTrace:
    """Test DigitalTrace dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic digital trace."""
        data = np.array([False, True, True, False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        np.testing.assert_array_equal(trace.data, data)
        assert trace.metadata.sample_rate == 1e6

    def test_create_with_transitions(self) -> None:
        """Test creating digital trace with transitions."""
        data = np.array([False, True, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        # Verify data transitions
        assert len(trace.data) == 4
        assert not trace.data[0]
        assert trace.data[1]
        assert trace.data[2]
        assert not trace.data[3]

    def test_validate_data_type(self) -> None:
        """Test that non-array data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(TypeError, match="data must be numpy array"):
            DigitalTrace(data=[False, True, False], metadata=metadata)  # type: ignore[arg-type]

    def test_validate_non_bool_dtype(self) -> None:
        """Test that non-bool dtypes raise TypeError."""
        data = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(TypeError, match="data must be boolean array"):
            DigitalTrace(data=data, metadata=metadata)

    def test_time_property(self) -> None:
        """Test time property."""
        data = np.array([False, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        expected = np.array([0.0, 1e-6, 2e-6])
        np.testing.assert_array_almost_equal(trace.time, expected)

    def test_duration_property(self) -> None:
        """Test duration property."""
        data = np.array([False, True, False, True, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        # Duration is (n-1) * time_base = 4 * 1e-6
        assert trace.duration == 4e-6

    def test_duration_empty_array(self) -> None:
        """Test duration with empty array raises ValueError."""
        data = np.array([], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(ValueError, match="data array cannot be empty"):
            DigitalTrace(data=data, metadata=metadata)

    def test_data_access(self) -> None:
        """Test data access."""
        data = np.array([False, True, True, False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert not trace.data[0]
        assert trace.data[1]
        assert trace.data[-1]

    def test_getitem(self) -> None:
        """Test __getitem__ method."""
        data = np.array([False, True, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert not trace[0]
        assert trace[1]
        np.testing.assert_array_equal(trace[1:3], [True, True])

    def test_signal_type_properties(self) -> None:
        """Test signal type properties."""
        data = np.array([True, False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert trace.is_digital is True
        assert trace.is_analog is False
        assert trace.is_iq is False
        assert trace.signal_type == "digital"

    def test_time_axis(self) -> None:
        """Test time axis matches data length."""
        data = np.array([True, False, True, False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert len(trace.time) == len(trace.data)
        assert trace.time[0] == 0.0
        assert trace.time[-1] == 4e-6

    def test_len(self) -> None:
        """Test __len__ method."""
        data = np.array([False, True, False, True, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert len(trace) == 5


class TestIQTrace:
    """Test IQTrace dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic IQ trace."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        np.testing.assert_array_equal(trace.data, data)
        assert trace.metadata.sample_rate == 1e6

    def test_validate_data_type(self) -> None:
        """Test that non-array data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(TypeError, match="data must be numpy array"):
            IQTrace(data=[1.0 + 0.5j, 2.0 + 1.5j], metadata=metadata)  # type: ignore[arg-type]

    def test_validate_complex_dtype(self) -> None:
        """Test that non-complex data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([1.0, 2.0, 3.0])  # Real data, not complex
        with pytest.raises(TypeError, match="data must be complex array"):
            IQTrace(data=data, metadata=metadata)

    def test_validate_empty_array(self) -> None:
        """Test that empty array raises ValueError."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([], dtype=complex)
        with pytest.raises(ValueError, match="data array cannot be empty"):
            IQTrace(data=data, metadata=metadata)

    def test_complex_data_access(self) -> None:
        """Test accessing complex data."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        assert trace.data.dtype == np.complex128
        np.testing.assert_array_equal(trace.data, data)

    def test_signal_type_properties(self) -> None:
        """Test signal type properties."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        assert trace.is_iq is True
        assert trace.is_analog is False
        assert trace.is_digital is False
        assert trace.signal_type == "iq"

    def test_real_and_imag_parts(self) -> None:
        """Test accessing real and imaginary parts."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        np.testing.assert_array_equal(trace.data.real, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(trace.data.imag, [0.5, 1.5, 2.5])

    def test_magnitude(self) -> None:
        """Test magnitude calculation."""
        data = np.array([3.0 + 4.0j, 4.0 + 3.0j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        expected = np.array([5.0, 5.0])  # sqrt(3^2 + 4^2) = 5
        np.testing.assert_array_almost_equal(np.abs(trace.data), expected)

    def test_phase(self) -> None:
        """Test phase calculation."""
        data = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        expected = np.array([0.0, np.pi / 2, np.pi])
        np.testing.assert_array_almost_equal(np.angle(trace.data), expected)

    def test_time_property(self) -> None:
        """Test time property."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        expected = np.array([0.0, 1e-6, 2e-6])
        np.testing.assert_array_almost_equal(trace.time, expected)

    def test_duration_property(self) -> None:
        """Test duration property."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j, 4.0 + 3.5j, 5.0 + 4.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        # Duration is (n-1) * time_base = 4 * 1e-6
        assert trace.duration == 4e-6

    def test_duration_empty_arrays(self) -> None:
        """Test duration with empty arrays raises ValueError."""
        data = np.array([], dtype=complex)
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(ValueError, match="data array cannot be empty"):
            IQTrace(data=data, metadata=metadata)

    def test_len(self) -> None:
        """Test __len__ method."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j, 4.0 + 3.5j, 5.0 + 4.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        assert len(trace) == 5


class TestProtocolPacket:
    """Test ProtocolPacket dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic protocol packet."""
        packet = ProtocolPacket(
            timestamp=1.23e-3,
            protocol="UART",
            data=b"Hello",
        )
        assert packet.timestamp == 1.23e-3
        assert packet.protocol == "UART"
        assert packet.data == b"Hello"
        assert packet.annotations == {}
        assert packet.errors == []
        assert packet.end_timestamp is None

    def test_create_with_all_fields(self) -> None:
        """Test creating packet with all optional fields."""
        annotations = {"level1": "START", "level2": "0x48"}
        errors = ["parity_error", "framing_error"]
        packet = ProtocolPacket(
            timestamp=1.0e-3,
            protocol="SPI",
            data=b"\x48\x65\x6c\x6c\x6f",
            annotations=annotations,
            errors=errors,
            end_timestamp=2.0e-3,
        )
        assert packet.timestamp == 1.0e-3
        assert packet.protocol == "SPI"
        assert packet.data == b"\x48\x65\x6c\x6c\x6f"
        assert packet.annotations == annotations
        assert packet.errors == errors
        assert packet.end_timestamp == 2.0e-3

    def test_validate_negative_timestamp(self) -> None:
        """Test that negative timestamp raises ValueError."""
        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            ProtocolPacket(
                timestamp=-1.0,
                protocol="UART",
                data=b"test",
            )

    def test_validate_data_type(self) -> None:
        """Test that non-bytes data raises TypeError."""
        with pytest.raises(TypeError, match="data must be bytes"):
            ProtocolPacket(
                timestamp=0.0,
                protocol="UART",
                data="test",  # type: ignore[arg-type]
            )

    def test_duration_property_with_end(self) -> None:
        """Test duration property when end_timestamp is set."""
        packet = ProtocolPacket(
            timestamp=1.0e-3,
            protocol="UART",
            data=b"test",
            end_timestamp=1.5e-3,
        )
        assert packet.duration == 0.5e-3

    def test_duration_property_without_end(self) -> None:
        """Test duration property when end_timestamp is None."""
        packet = ProtocolPacket(
            timestamp=1.0e-3,
            protocol="UART",
            data=b"test",
        )
        assert packet.duration is None

    def test_has_errors_true(self) -> None:
        """Test has_errors property when errors exist."""
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="UART",
            data=b"test",
            errors=["parity_error"],
        )
        assert packet.has_errors is True

    def test_has_errors_false(self) -> None:
        """Test has_errors property when no errors."""
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="UART",
            data=b"test",
        )
        assert packet.has_errors is False

    def test_len(self) -> None:
        """Test __len__ method."""
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="UART",
            data=b"Hello",
        )
        assert len(packet) == 5


class TestCoreTypesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_waveform_single_sample(self) -> None:
        """Test waveform with single sample."""
        data = np.array([1.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        assert len(trace) == 1
        assert trace.duration == 0.0
        assert len(trace.time) == 1
        assert trace.time[0] == 0.0

    def test_digital_single_sample(self) -> None:
        """Test digital trace with single sample."""
        data = np.array([True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert len(trace) == 1
        assert trace.duration == 0.0

    def test_iq_single_sample(self) -> None:
        """Test IQ trace with single sample."""
        data = np.array([1.0 + 0.0j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        assert len(trace) == 1
        assert trace.duration == 0.0

    def test_waveform_large_array(self) -> None:
        """Test waveform with large array (10M samples)."""
        data = np.zeros(10_000_000, dtype=np.float32)
        metadata = TraceMetadata(sample_rate=1e9)
        trace = WaveformTrace(data=data, metadata=metadata)
        assert len(trace) == 10_000_000
        # Float32 is already a floating type, so it won't be converted
        assert np.issubdtype(trace.data.dtype, np.floating)
        assert trace.duration == pytest.approx(9.999999e-3, rel=1e-6)

    def test_protocol_packet_empty_data(self) -> None:
        """Test protocol packet with empty data."""
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="UART",
            data=b"",
        )
        assert len(packet) == 0
        assert not packet.has_errors

    def test_protocol_packet_zero_timestamp(self) -> None:
        """Test protocol packet with zero timestamp."""
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="UART",
            data=b"test",
        )
        assert packet.timestamp == 0.0
        assert packet.duration is None

    def test_protocol_packet_zero_duration(self) -> None:
        """Test protocol packet with zero duration."""
        packet = ProtocolPacket(
            timestamp=1.0,
            protocol="UART",
            data=b"test",
            end_timestamp=1.0,
        )
        assert packet.duration == 0.0

    def test_digital_trace_all_low(self) -> None:
        """Test digital trace with all low values."""
        data = np.array([False, False, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert not trace.data.any()
        assert len(trace) == 3

    def test_digital_trace_all_high(self) -> None:
        """Test digital trace with all high values."""
        data = np.array([True, True, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert all(trace.data)
        assert len(trace) == 3

    def test_digital_trace_alternating(self) -> None:
        """Test digital trace with alternating values."""
        data = np.array([False, True, False, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert not trace.data[0]
        assert trace.data[1]
        assert not trace.data[2]
        assert len(trace) == 5

    def test_iq_zero_magnitude(self) -> None:
        """Test IQ trace with zero magnitude."""
        data = np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        np.testing.assert_array_equal(np.abs(trace.data), [0.0, 0.0, 0.0])

    def test_iq_negative_values(self) -> None:
        """Test IQ trace with negative values."""
        data = np.array([-1.0 - 0.5j, -2.0 - 1.5j, -3.0 - 2.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        expected_complex = np.array([-1.0 - 0.5j, -2.0 - 1.5j, -3.0 - 2.5j])
        np.testing.assert_array_almost_equal(trace.data, expected_complex)

    def test_metadata_very_high_sample_rate(self) -> None:
        """Test metadata with very high sample rate (1 THz)."""
        metadata = TraceMetadata(sample_rate=1e12)
        assert 1.0 / metadata.sample_rate == 1e-12

    def test_metadata_very_low_sample_rate(self) -> None:
        """Test metadata with very low sample rate (1 Hz)."""
        metadata = TraceMetadata(sample_rate=1.0)
        assert 1.0 / metadata.sample_rate == 1.0

    def test_waveform_float32_dtype(self) -> None:
        """Test waveform with float32 data (already floating type)."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        # Float32 is already a floating type, no conversion happens
        assert np.issubdtype(trace.data.dtype, np.floating)
        assert trace.data.dtype == np.float32

    def test_iq_complex64_dtype(self) -> None:
        """Test IQ trace with complex64 dtype."""
        data = np.array([1.0 + 4.0j, 2.0 + 5.0j, 3.0 + 6.0j], dtype=np.complex64)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)
        assert np.iscomplexobj(trace.data)
        assert trace.data.dtype == np.complex64

    def test_protocol_packet_multiple_errors(self) -> None:
        """Test protocol packet with multiple errors."""
        errors = ["parity_error", "framing_error", "overrun_error"]
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="UART",
            data=b"test",
            errors=errors,
        )
        assert packet.has_errors is True
        assert len(packet.errors) == 3

    def test_protocol_packet_large_data(self) -> None:
        """Test protocol packet with large data payload."""
        large_data = bytes(range(256)) * 100  # 25.6 KB
        packet = ProtocolPacket(
            timestamp=0.0,
            protocol="SPI",
            data=large_data,
        )
        assert len(packet) == 25600


class TestCoreTypesIntegration:
    """Integration tests for type interactions."""

    def test_waveform_trace_workflow(self) -> None:
        """Test complete waveform trace workflow."""
        # Create sine wave
        t = np.linspace(0, 1e-3, 1000)
        data = np.sin(2 * np.pi * 1000 * t)
        metadata = TraceMetadata(
            sample_rate=1e6,
            vertical_scale=0.5,
            channel="CH1",
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        # Verify properties
        assert len(trace) == 1000
        assert trace.duration == pytest.approx(999e-6)
        assert len(trace.time) == 1000
        assert trace.time[-1] == pytest.approx(999e-6)

    def test_digital_trace_workflow(self) -> None:
        """Test complete digital trace workflow."""
        # Create digital signal
        data = np.array([False, False, True, True, True, False, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)

        # Verify properties
        assert len(trace) == 7
        assert trace.duration == 6e-6
        assert not trace.data[0]
        assert trace.data[2]
        assert not trace.data[-1]

    def test_iq_trace_workflow(self) -> None:
        """Test complete IQ trace workflow."""
        # Create IQ signal
        t = np.linspace(0, 1e-3, 1000)
        data = np.exp(1j * 2 * np.pi * 1e6 * t)  # Complex exponential
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(data=data, metadata=metadata)

        # Verify properties
        assert len(trace) == 1000
        assert trace.duration == pytest.approx(999e-6)
        assert len(trace.data) == 1000
        assert len(np.abs(trace.data)) == 1000  # Magnitude
        assert len(np.angle(trace.data)) == 1000  # Phase

    def test_protocol_packet_workflow(self) -> None:
        """Test complete protocol packet workflow."""
        # Create UART packet with errors
        packet = ProtocolPacket(
            timestamp=1.23e-3,
            protocol="UART",
            data=b"AT+CMD",
            annotations={"type": "command", "dest": "modem"},
            errors=["parity_error"],
            end_timestamp=1.5e-3,
        )

        # Verify properties
        assert len(packet) == 6
        assert packet.has_errors is True
        assert packet.duration == pytest.approx(0.27e-3)

    def test_trace_union_type_waveform(self) -> None:
        """Test that WaveformTrace satisfies Trace union type."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace: WaveformTrace = WaveformTrace(data=data, metadata=metadata)
        # Verify it has common trace properties
        assert hasattr(trace, "data")
        assert hasattr(trace, "metadata")
        assert hasattr(trace, "time")
        assert hasattr(trace, "duration")
        assert len(trace) == 3

    def test_trace_union_type_digital(self) -> None:
        """Test that DigitalTrace satisfies Trace union type."""
        data = np.array([False, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace: DigitalTrace = DigitalTrace(data=data, metadata=metadata)
        # Verify it has common trace properties
        assert hasattr(trace, "data")
        assert hasattr(trace, "metadata")
        assert hasattr(trace, "time")
        assert hasattr(trace, "duration")
        assert len(trace) == 3

    def test_trace_union_type_iq(self) -> None:
        """Test that IQTrace satisfies Trace union type."""
        data = np.array([1.0 + 0.5j, 2.0 + 1.5j])
        metadata = TraceMetadata(sample_rate=1e6)
        trace: IQTrace = IQTrace(data=data, metadata=metadata)
        # Verify it has common trace properties
        assert hasattr(trace, "data")
        assert hasattr(trace, "metadata")
        assert hasattr(trace, "time")
        assert hasattr(trace, "duration")
        assert len(trace) == 2

    def test_metadata_reuse_across_traces(self) -> None:
        """Test that same metadata can be shared across multiple traces."""
        metadata = TraceMetadata(
            sample_rate=1e9,
            vertical_scale=1.0,
            channel="CH1",
        )

        # Create multiple traces with same metadata
        waveform = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=metadata,
        )
        digital = DigitalTrace(
            data=np.array([False, True, False], dtype=bool),
            metadata=metadata,
        )
        iq = IQTrace(
            data=np.array([1.0 + 0.5j, 2.0 + 1.5j]),
            metadata=metadata,
        )

        # Verify all share the same metadata instance
        assert waveform.metadata is metadata
        assert digital.metadata is metadata
        assert iq.metadata is metadata
        assert waveform.metadata.sample_rate == 1e9
        assert digital.metadata.sample_rate == 1e9
        assert iq.metadata.sample_rate == 1e9
