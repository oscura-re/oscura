"""Tests for core data types.

Tests the fundamental data structures (CORE-001-004).
"""

from __future__ import annotations

from datetime import UTC, datetime

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
        assert metadata.acquisition_time is None
        assert metadata.trigger_info is None
        assert metadata.source_file is None
        assert metadata.channel_name is None

    def test_create_with_all_fields(self) -> None:
        """Test creating metadata with all optional fields."""
        acq_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        trigger = {"type": "edge", "level": 1.5}
        metadata = TraceMetadata(
            sample_rate=1e9,
            vertical_scale=0.5,
            vertical_offset=0.1,
            acquisition_time=acq_time,
            trigger_info=trigger,
            source_file="/path/to/data.bin",
            channel_name="CH1",
        )
        assert metadata.sample_rate == 1e9
        assert metadata.vertical_scale == 0.5
        assert metadata.vertical_offset == 0.1
        assert metadata.acquisition_time == acq_time
        assert metadata.trigger_info == trigger
        assert metadata.source_file == "/path/to/data.bin"
        assert metadata.channel_name == "CH1"

    def test_validate_positive_sample_rate(self) -> None:
        """Test that negative sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TraceMetadata(sample_rate=-100.0)

    def test_validate_zero_sample_rate(self) -> None:
        """Test that zero sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TraceMetadata(sample_rate=0.0)

    def test_time_base_property(self) -> None:
        """Test time_base property calculation."""
        metadata = TraceMetadata(sample_rate=1e6)
        assert metadata.time_base == 1e-6

        metadata2 = TraceMetadata(sample_rate=1e9)
        assert metadata2.time_base == 1e-9


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
        with pytest.raises(TypeError, match="data must be a numpy array"):
            WaveformTrace(data=[1.0, 2.0, 3.0], metadata=metadata)  # type: ignore[arg-type]

    def test_auto_convert_to_float(self) -> None:
        """Test that non-float dtypes are converted to float64."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        assert trace.data.dtype == np.float64
        np.testing.assert_array_equal(trace.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_time_vector_property(self) -> None:
        """Test time_vector property."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        expected = np.array([0.0, 1e-6, 2e-6])
        np.testing.assert_array_almost_equal(trace.time_vector, expected)

    def test_duration_property(self) -> None:
        """Test duration property."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 samples
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        # Duration is (n-1) * time_base = 4 * 1e-6
        assert trace.duration == 4e-6

    def test_duration_empty_array(self) -> None:
        """Test duration with empty array."""
        data = np.array([])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        assert trace.duration == 0.0

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
        assert trace.edges is None

    def test_create_with_edges(self) -> None:
        """Test creating digital trace with edge information."""
        data = np.array([False, True, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        edges = [(1e-6, True), (3e-6, False)]  # Rising at 1us, falling at 3us
        trace = DigitalTrace(data=data, metadata=metadata, edges=edges)
        assert trace.edges == edges

    def test_validate_data_type(self) -> None:
        """Test that non-array data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(TypeError, match="data must be a numpy array"):
            DigitalTrace(data=[False, True, False], metadata=metadata)  # type: ignore[arg-type]

    def test_auto_convert_to_bool(self) -> None:
        """Test that non-bool dtypes are converted to bool."""
        data = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert trace.data.dtype == np.bool_
        expected = np.array([False, True, True, False, True], dtype=bool)
        np.testing.assert_array_equal(trace.data, expected)

    def test_time_vector_property(self) -> None:
        """Test time_vector property."""
        data = np.array([False, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        expected = np.array([0.0, 1e-6, 2e-6])
        np.testing.assert_array_almost_equal(trace.time_vector, expected)

    def test_duration_property(self) -> None:
        """Test duration property."""
        data = np.array([False, True, False, True, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        # Duration is (n-1) * time_base = 4 * 1e-6
        assert trace.duration == 4e-6

    def test_duration_empty_array(self) -> None:
        """Test duration with empty array."""
        data = np.array([], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert trace.duration == 0.0

    def test_rising_edges_property(self) -> None:
        """Test rising_edges property."""
        data = np.array([False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        edges = [(1e-6, True), (3e-6, False), (5e-6, True)]
        trace = DigitalTrace(data=data, metadata=metadata, edges=edges)
        assert trace.rising_edges == [1e-6, 5e-6]

    def test_rising_edges_none(self) -> None:
        """Test rising_edges when edges is None."""
        data = np.array([False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert trace.rising_edges == []

    def test_falling_edges_property(self) -> None:
        """Test falling_edges property."""
        data = np.array([True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        edges = [(1e-6, True), (3e-6, False), (5e-6, True), (7e-6, False)]
        trace = DigitalTrace(data=data, metadata=metadata, edges=edges)
        assert trace.falling_edges == [3e-6, 7e-6]

    def test_falling_edges_none(self) -> None:
        """Test falling_edges when edges is None."""
        data = np.array([True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert trace.falling_edges == []

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
        i_data = np.array([1.0, 2.0, 3.0])
        q_data = np.array([0.5, 1.5, 2.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        np.testing.assert_array_equal(trace.i_data, i_data)
        np.testing.assert_array_equal(trace.q_data, q_data)
        assert trace.metadata.sample_rate == 1e6

    def test_validate_i_data_type(self) -> None:
        """Test that non-array i_data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        q_data = np.array([0.5, 1.5])
        with pytest.raises(TypeError, match="i_data must be a numpy array"):
            IQTrace(i_data=[1.0, 2.0], q_data=q_data, metadata=metadata)  # type: ignore[arg-type]

    def test_validate_q_data_type(self) -> None:
        """Test that non-array q_data raises TypeError."""
        metadata = TraceMetadata(sample_rate=1e6)
        i_data = np.array([1.0, 2.0])
        with pytest.raises(TypeError, match="q_data must be a numpy array"):
            IQTrace(i_data=i_data, q_data=[0.5, 1.5], metadata=metadata)  # type: ignore[arg-type]

    def test_validate_length_mismatch(self) -> None:
        """Test that mismatched I/Q lengths raise ValueError."""
        metadata = TraceMetadata(sample_rate=1e6)
        i_data = np.array([1.0, 2.0, 3.0])
        q_data = np.array([0.5, 1.5])  # Different length
        with pytest.raises(ValueError, match="I and Q data must have same length"):
            IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)

    def test_auto_convert_i_to_float(self) -> None:
        """Test that non-float i_data is converted to float64."""
        i_data = np.array([1, 2, 3], dtype=np.int32)
        q_data = np.array([0.5, 1.5, 2.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        assert trace.i_data.dtype == np.float64
        np.testing.assert_array_equal(trace.i_data, [1.0, 2.0, 3.0])

    def test_auto_convert_q_to_float(self) -> None:
        """Test that non-float q_data is converted to float64."""
        i_data = np.array([1.0, 2.0, 3.0])
        q_data = np.array([1, 2, 3], dtype=np.int32)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        assert trace.q_data.dtype == np.float64
        np.testing.assert_array_equal(trace.q_data, [1.0, 2.0, 3.0])

    def test_complex_data_property(self) -> None:
        """Test complex_data property."""
        i_data = np.array([1.0, 2.0, 3.0])
        q_data = np.array([0.5, 1.5, 2.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        expected = np.array([1.0 + 0.5j, 2.0 + 1.5j, 3.0 + 2.5j])
        np.testing.assert_array_almost_equal(trace.complex_data, expected)

    def test_magnitude_property(self) -> None:
        """Test magnitude property."""
        i_data = np.array([3.0, 4.0])
        q_data = np.array([4.0, 3.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        expected = np.array([5.0, 5.0])  # sqrt(3^2 + 4^2) = 5
        np.testing.assert_array_almost_equal(trace.magnitude, expected)

    def test_phase_property(self) -> None:
        """Test phase property."""
        i_data = np.array([1.0, 0.0, -1.0])
        q_data = np.array([0.0, 1.0, 0.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        expected = np.array([0.0, np.pi / 2, np.pi])
        np.testing.assert_array_almost_equal(trace.phase, expected)

    def test_time_vector_property(self) -> None:
        """Test time_vector property."""
        i_data = np.array([1.0, 2.0, 3.0])
        q_data = np.array([0.5, 1.5, 2.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        expected = np.array([0.0, 1e-6, 2e-6])
        np.testing.assert_array_almost_equal(trace.time_vector, expected)

    def test_duration_property(self) -> None:
        """Test duration property."""
        i_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_data = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        # Duration is (n-1) * time_base = 4 * 1e-6
        assert trace.duration == 4e-6

    def test_duration_empty_arrays(self) -> None:
        """Test duration with empty arrays."""
        i_data = np.array([])
        q_data = np.array([])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        assert trace.duration == 0.0

    def test_len(self) -> None:
        """Test __len__ method."""
        i_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_data = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
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
        assert len(trace.time_vector) == 1
        assert trace.time_vector[0] == 0.0

    def test_digital_single_sample(self) -> None:
        """Test digital trace with single sample."""
        data = np.array([True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)
        assert len(trace) == 1
        assert trace.duration == 0.0

    def test_iq_single_sample(self) -> None:
        """Test IQ trace with single sample."""
        i_data = np.array([1.0])
        q_data = np.array([0.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
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

    def test_digital_edges_empty_list(self) -> None:
        """Test digital trace with empty edges list."""
        data = np.array([False, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata, edges=[])
        assert trace.rising_edges == []
        assert trace.falling_edges == []

    def test_digital_edges_all_rising(self) -> None:
        """Test digital trace with only rising edges."""
        data = np.array([False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        edges = [(1e-6, True), (2e-6, True), (3e-6, True)]
        trace = DigitalTrace(data=data, metadata=metadata, edges=edges)
        assert len(trace.rising_edges) == 3
        assert len(trace.falling_edges) == 0

    def test_digital_edges_all_falling(self) -> None:
        """Test digital trace with only falling edges."""
        data = np.array([True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        edges = [(1e-6, False), (2e-6, False), (3e-6, False)]
        trace = DigitalTrace(data=data, metadata=metadata, edges=edges)
        assert len(trace.rising_edges) == 0
        assert len(trace.falling_edges) == 3

    def test_iq_zero_magnitude(self) -> None:
        """Test IQ trace with zero magnitude."""
        i_data = np.array([0.0, 0.0, 0.0])
        q_data = np.array([0.0, 0.0, 0.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        np.testing.assert_array_equal(trace.magnitude, [0.0, 0.0, 0.0])

    def test_iq_negative_values(self) -> None:
        """Test IQ trace with negative values."""
        i_data = np.array([-1.0, -2.0, -3.0])
        q_data = np.array([-0.5, -1.5, -2.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        expected_complex = np.array([-1.0 - 0.5j, -2.0 - 1.5j, -3.0 - 2.5j])
        np.testing.assert_array_almost_equal(trace.complex_data, expected_complex)

    def test_metadata_very_high_sample_rate(self) -> None:
        """Test metadata with very high sample rate (1 THz)."""
        metadata = TraceMetadata(sample_rate=1e12)
        assert metadata.time_base == 1e-12

    def test_metadata_very_low_sample_rate(self) -> None:
        """Test metadata with very low sample rate (1 Hz)."""
        metadata = TraceMetadata(sample_rate=1.0)
        assert metadata.time_base == 1.0

    def test_waveform_float32_dtype(self) -> None:
        """Test waveform with float32 data (already floating type)."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        # Float32 is already a floating type, no conversion happens
        assert np.issubdtype(trace.data.dtype, np.floating)
        assert trace.data.dtype == np.float32

    def test_iq_both_integer_conversion(self) -> None:
        """Test IQ trace with both I and Q as integers."""
        i_data = np.array([1, 2, 3], dtype=np.int32)
        q_data = np.array([4, 5, 6], dtype=np.int32)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        assert trace.i_data.dtype == np.float64
        assert trace.q_data.dtype == np.float64

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
            channel_name="CH1",
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        # Verify properties
        assert len(trace) == 1000
        assert trace.duration == pytest.approx(999e-6)
        assert len(trace.time_vector) == 1000
        assert trace.time_vector[-1] == pytest.approx(999e-6)

    def test_digital_trace_workflow(self) -> None:
        """Test complete digital trace workflow."""
        # Create digital signal with edges
        data = np.array([False, False, True, True, True, False, False], dtype=bool)
        edges = [(2e-6, True), (5e-6, False)]
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata, edges=edges)

        # Verify properties
        assert len(trace) == 7
        assert trace.duration == 6e-6
        assert trace.rising_edges == [2e-6]
        assert trace.falling_edges == [5e-6]

    def test_iq_trace_workflow(self) -> None:
        """Test complete IQ trace workflow."""
        # Create IQ signal
        t = np.linspace(0, 1e-3, 1000)
        i_data = np.cos(2 * np.pi * 1e6 * t)
        q_data = np.sin(2 * np.pi * 1e6 * t)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)

        # Verify properties
        assert len(trace) == 1000
        assert trace.duration == pytest.approx(999e-6)
        assert len(trace.complex_data) == 1000
        assert len(trace.magnitude) == 1000
        assert len(trace.phase) == 1000

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
        assert hasattr(trace, "time_vector")
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
        assert hasattr(trace, "time_vector")
        assert hasattr(trace, "duration")
        assert len(trace) == 3

    def test_trace_union_type_iq(self) -> None:
        """Test that IQTrace satisfies Trace union type."""
        i_data = np.array([1.0, 2.0])
        q_data = np.array([0.5, 1.5])
        metadata = TraceMetadata(sample_rate=1e6)
        trace: IQTrace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)
        # Verify it has common trace properties
        assert hasattr(trace, "i_data")
        assert hasattr(trace, "q_data")
        assert hasattr(trace, "metadata")
        assert hasattr(trace, "time_vector")
        assert hasattr(trace, "duration")
        assert len(trace) == 2

    def test_metadata_reuse_across_traces(self) -> None:
        """Test that same metadata can be shared across multiple traces."""
        metadata = TraceMetadata(
            sample_rate=1e9,
            vertical_scale=1.0,
            channel_name="CH1",
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
            i_data=np.array([1.0, 2.0]),
            q_data=np.array([0.5, 1.5]),
            metadata=metadata,
        )

        # Verify all share the same metadata instance
        assert waveform.metadata is metadata
        assert digital.metadata is metadata
        assert iq.metadata is metadata
        assert waveform.metadata.sample_rate == 1e9
        assert digital.metadata.sample_rate == 1e9
        assert iq.metadata.sample_rate == 1e9
